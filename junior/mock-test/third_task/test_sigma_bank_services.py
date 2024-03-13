from unittest.mock import Mock, patch
from sigma_bank_services import check_balance, transfer_money, UserNotFoundException, InsufficientBalanceException

# Создаем mock-объект для BankingAPI
mock_api = Mock()


def test_check_balance_success():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Устанавливаем возвращаемое значение для метода get_balance
        mock_api.get_balance.return_value = 1500

        # Тестируем функцию check_balance для любого user_id
        balance = check_balance("User123")

        # Проверяем результат и вызов метода
        assert balance == 1500
        mock_api.get_balance.assert_called_once()

        # Сбрасываем состояние mock-объекта
        mock_api.reset_mock()


def test_check_balance_user_not_found():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Устанавливаем возвращаемое исключение для метода get_balance
        mock_api.get_balance.side_effect = UserNotFoundException("User not found!")

        # Тестируем check_balance для любого user_id и утверждаем
        try:
            check_balance("User123")
        except UserNotFoundException:
            assert True
        else:
            assert False, "UserNotFoundException not raised"

        # Сбрасываем состояние mock-объекта
        mock_api.reset_mock()


def test_transfer_money_success():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Устанавливаем возвращаемое значение для метода initiate_transfer
        mock_api.initiate_transfer.return_value = "Transfer Successful"

        # Тестируем функцию transfer_money
        status = transfer_money("User123", "User456", 200)

        # Проверяем результат и вызовы методов
        assert status == "Transfer Successful"
        mock_api.initiate_transfer.assert_called_once_with("User123", "User456", 200)
        # Сбрасываем состояние mock-объекта
        mock_api.reset_mock()


def test_transfer_money_insufficient_balance():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        mock_api.get_balance.return_value = 100
        mock_api.initiate_transfer.side_effect = InsufficientBalanceException("Insufficient balance")

        try:
            transfer_money("User123", "User456", 200)
        except InsufficientBalanceException:
            assert True
        else:
            assert False, "InsufficientBalanceException not raised"

        mock_api.reset_mock()


def test_transfer_money_user_not_found():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        mock_api.initiate_transfer.side_effect = UserNotFoundException("User not found!")

        try:
            transfer_money("User123", "User4563", 200)
        except UserNotFoundException:
            assert True
        else:
            assert False, "UserNotFoundException not raised"

    mock_api.reset_mock()
